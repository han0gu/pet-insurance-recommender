from langchain_core.documents import Document

chunk = Document(
    page_content=('- 중 높은 지급률을 적용함을 원칙으로 한다. 그러나 각 신체부위별 판정기준\n'
 '140 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 에서 별도로 정한 경우에는 그 기준에 따른다.\n'
 '- 3) 하나의 장해가 다른 장해와 통상 파생하는 관계에 있는 경우에는 그중 높은\n'
 '- 지급률만을 적용하며, 하나의 장해로 둘 이상의 파생장해가 발생하는 경우\n'
 '- 각 파생장해의 지급률을 합산한 지급률과 최초 장해의 지급률을 비교하여 그\n'
 '- 중 높은 지급률을 적용한다.\n'
 '- 4) 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기능을 상실하여 인공심박'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000835',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
