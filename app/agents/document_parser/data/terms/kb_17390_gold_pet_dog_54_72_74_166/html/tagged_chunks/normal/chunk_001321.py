from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같습니다)하던 중 발생한 급격하고도 우연한<br>외래의 상해사고를 직접적인 원인으로 보험계약에서 정한 보험금 지급사유가 '
 '발생<br>한 경우에 보험금을 지급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001321',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
