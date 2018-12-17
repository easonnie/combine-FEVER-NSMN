#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from utils.fever_db import convert_brc
from chaonan_src._utils.wiki_pageview_utils import WikiPageviews
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral


__author__ = ['chaonan99']


class ItemRuleRawPageview(ItemRuleBuilderSpiral):
    """docstring for ItemRuleRawPageview"""
    def __init__(self):
        # super(ItemRuleRawPageview, self).__init__()
        pass

    def pageview_rule(self):
        """Assign high priority to frequently viewed pages
        """
        if not hasattr(self, 'wiki_pv'):
            print("Reload wiki pageview dict")
            self.wiki_pv = WikiPageviews()

        item = self.item
        docid_groups = [[i[0] for i in it] \
                        for _, it in item['structured_docids'].items()]

        for key, group_prio_docids in item['structured_docids'].items():
            group_docids = [it[0] for it in group_prio_docids]
            all_scores = map(lambda x: self.wiki_pv[convert_brc(x)],
                             group_docids)
            all_scores = np.array(list(all_scores))
            prios = np.argsort(all_scores)[::-1]
            new_gpd = []
            for i, p in enumerate(prios):
                # new_gpd.append((group_prio_docids[p][0],
                #                 group_prio_docids[p][1] + \
                #                     max(1.0 - i*0.2, 0)))

                new_gpd.append((group_prio_docids[p][0],
                                int(all_scores[p])))
            item['structured_docids'][key] = new_gpd

        try:
            finded_keys = item['structured_docids'].values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)
            item['prioritized_docids'] = list(finded_keys)
        except Exception as e:
            from IPython import embed; embed(); import os; os._exit(1)
        return self

    def rules(self):
        return lambda x: self.initialize_item(x)\
                             .exact_match_rule()\
                             .singularize_rule()\
                             .eliminate_articles_rule()\
                             .pageview_rule()\


def main():
    pass


if __name__ == '__main__':
    main()


