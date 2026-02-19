from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 반려동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증상을 포함합니다. 다만, 보험기간 중 '
 '최초로 발견된 경우에는 해당 보험 기간에 한하여 보상합니다.) 2. 다음 정한 질병 및 이에 기인하는 질병(다만, 질병의 발생일로부터 '
 '과거 1년 이내의 동물병원 예방접종 기록이 있는 경우에는 보상합니다.)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
