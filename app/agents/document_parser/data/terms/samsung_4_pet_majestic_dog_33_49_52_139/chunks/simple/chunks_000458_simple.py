from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 개별사항\n'
 '제 1조 (보험금의 지급사유)\n'
 '① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합니다) 중 일상생활 중에 다음 각 호에 정하는 '
 '강력범죄에 의하여 사망하거나 신체 (의수, 의족, 의안, 의치 등 신체보조장구는 제외하나, 인공장기나 부분 의치 등 신체 에 이식되어 그 '
 '기능을 대신할 경우는 포함합니다)에 피해가 발생하였을 경우 아래에 정한 금액을 강력범죄피해보장(범죄유형별) 보험금으로 보험수익자에게 '
 '지급합니다.\n'
 '구 분 | 지급금액\n'
 '살 인 | 1,000만원\n'
 '강 간 | 500만원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
