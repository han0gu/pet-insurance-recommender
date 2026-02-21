from langchain_core.documents import Document

chunk = Document(
    page_content=('| 용 어 풀 이 보장개시일 회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료 등을 받은 날을 말하나, 회사가 승낙하기 '
 '전이라도 청약과 함께 제1회 보험료 등을 받은 경우 에는 제1회 보험료 등을 받은 날을 말합니다. 또한, 보장개시일을 계약일로 봅 | 용 '
 '어 풀 이 보장개시일 회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료 등을 받은 날을 말하나, 회사가 승낙하기 전이라도 '
 '청약과 함께 제1회 보험료 등을 받은 경우 에는 제1회 보험료 등을 받은 날을 말합니다. 또한, 보장개시일을 계약일로 봅 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000512',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
