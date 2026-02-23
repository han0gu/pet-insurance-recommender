from langchain_core.documents import Document

chunk = Document(
    page_content=('| 과되지 않은 보험료를 말합니다. 단, 사업방법서에 따라 회사가 보험료를 할인한 경우 할인금액은 차감합니다. 예) 1년치 '
 '보험료(연납)을 받은 후 6개월이 경과했다면, 6개월(미경과 기간)에 대응하는 것으로 미경과보험료라고 합니다. | 과되지 않은 보험료를 '
 '말합니다. 단, 사업방법서에 따라 회사가 보험료를 할인한 경우 할인금액은 차감합니다. 예) 1년치 보험료(연납)을 받은 후 6개월이 '
 '경과했다면, 6개월(미경과 기간)에 대응하는 것으로 미경과보험료라고 합니다. | 과되지 않은 보험료를 말합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
