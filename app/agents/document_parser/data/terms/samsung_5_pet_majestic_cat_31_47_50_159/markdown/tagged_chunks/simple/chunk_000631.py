from langchain_core.documents import Document

chunk = Document(
    page_content=('도액 중 가장 큰 보상한도액을 적용하며 아래 특별약관 중 가장 큰 보상한도액에 해\n'
 '당하는 특별약관 이외의 보험금은 지급되지 않습니다. 단, 아래의 특별약관에서 정한\n'
 '연간 보상횟수가 초과된 경우에는 보험금의 지급사유가 발생하지 않은 것으로 봅니다- 1. 반려묘 의료비 확대보장(이물제거 '
 '특정처치)(연간2회한)(재가입형)\n'
 '- 2. 반려묘 의료비 확대보장(MRI,CT)(연간1회한)(재가입형)\n'
 '# 제2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000631',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
