from langchain_core.documents import Document

chunk = Document(
    page_content=('한 특별약관에서 정한 보험금의 지급사유가 발생(특별약관별 면책기간과 연간보상한 도 및 연간보상횟수를 고려한 발생여부)한 경우에는 아래 '
 '특별약관에서 정한 보상한 도액 중 가장 큰 보상한도액을 적용하며 아래 특별약관 중 가장 큰 보상한도액에 해 당하는 특별약관 이외의 '
 '보험금은 지급되지 않습니다. 단, 아래의 특별약관에서 정한 연간 보상횟수가 초과된 경우에는 보험금의 지급사유가 발생하지 않은 것으로 '
 '봅니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000749',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
