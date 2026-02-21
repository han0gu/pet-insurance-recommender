from langchain_core.documents import Document

chunk = Document(
    page_content=('- 가입에 관한사항) 제5항에 따라 보험계약이 연장된 경우에는 보장개시일(책임개시일)\n'
 '- 은 이 특별약관의 보험계약일로 봅니다.\n'
 '- ⑨ 제4항의 보상한도액에도 불구하고 동일한 날에 아래의 특별약관 중 피보험자가 가입\n'
 '- 117 -한 특별약관에서 정한 보험금의 지급사유가 발생(특별약관별 면책기간과 연간보상한\n'
 '도 및 연간보상횟수를 고려한 발생여부)한 경우에는 아래 특별약관에서 정한 보상한\n'
 '도액 중 가장 큰 보상한도액을 적용하며 아래 특별약관 중 가장 큰 보상한도액에 해'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000630',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
