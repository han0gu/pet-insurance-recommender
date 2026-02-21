from langchain_core.documents import Document

chunk = Document(
    page_content=('그중 높은<br>지급률만을 적용하며, 하나의 장해로 둘 이상의 파생장해가 발생하는 경우<br>각 파생장해의 지급률을 합산한 지급률과 최초 '
 '장해의 지급률을 비교하여 그<br>중 높은 지급률을 적용한다.<br>4) 의학적으로 뇌사판정을 받고 호흡기능과 심장박동기능을 상실하여 '
 '인공심박<br>동기 등 장치에 의존하여 생명을 연장하고 있는 뇌사상태는 장해의 판정대상<br>에 포함되지 않는다'),
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
 'indexing': {'chunk_id': 'chunk_001475',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
