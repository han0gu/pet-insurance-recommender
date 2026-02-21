from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 반려동물장례비용지원금(실손)(30일면책)(강아지)【갱신계약】\n'
 '(【갱신계약】은 자동갱신으로 운영합니다)제1조(보험금의 지급사유)# \uf000 회사는 보험증권에기재된 반려동물이 이 특별약관의 보험기간 '
 '중 반려동물장례비- 용지원금의 보장개시일(이하 반려동물장례비용지원금보장개시일이라 합니다) 이\n'
 '- 후에 사망한 경우 동물장묘업체에서 제공하는 반려동물 장례서비스를 이용함으로\n'
 '- 써 장례 당일 발생한 총 장례비용을 제5항에 따라 반려동물장례비용지원금으로 보\n'
 '- 험수익자에게 지급합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
