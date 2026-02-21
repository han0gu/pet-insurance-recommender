from langchain_core.documents import Document

chunk = Document(
    page_content=('- 은 지급률을 적용함을 원칙으로 한다. 그러나 각 신체부위별 판정기준에서 별도\n'
 '# 로 정한 경우에는 그 기준에 따른다.- 다. 하나의 장해가 다른 장해와 통상 파생하는 관계에 있는 경우에는 그중 높은 지급\n'
 '- 률만을 적용하며, 하나의 장해로 둘 이상의 파생장해가 발생하는 경우 각 파생장\n'
 '- 해의 지급률을 합산한 지급률과 최초 장해의 지급률을 비교하여 그 중 높은 지급\n'
 '- 률을 적용한다.\n'
 '- 라. 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기능을 상실하여 인공심박동기'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000732',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
