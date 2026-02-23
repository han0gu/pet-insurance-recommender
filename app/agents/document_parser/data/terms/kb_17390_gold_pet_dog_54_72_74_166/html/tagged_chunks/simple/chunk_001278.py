from langchain_core.documents import Document

chunk = Document(
    page_content=('. 해<br>그러나, 동일한 질병에 대한 입원이라도 반려동물 위탁비용이 지급된 최종 입원 및<br>의 퇴원일로부터 180일이 지나서 '
 '개시한 입원은 새로운 입원으로 봅니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001278',
              'chunk_char_len': 94,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
