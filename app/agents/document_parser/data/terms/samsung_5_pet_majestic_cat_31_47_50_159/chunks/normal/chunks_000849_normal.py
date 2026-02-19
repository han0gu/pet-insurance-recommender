from langchain_core.documents import Document

chunk = Document(
    page_content=('제1조 (적용대상)\n'
 '이 특별약관은 보험계약자(이하 「계약자」 라 합니다), 피보험자 및 보험수익자가 모두 동 일한 보험계약(특별약관이 부가된 경우에는 그 '
 '특별약관을 포함합니다. 이하 「보험계약」 이라 합니다)에 적용합니다.\n'
 '제2조 (특별약관의 체결 및 소멸)\n'
 '① 이 특별약관은 계약자의 청약과 보험회사(이하 「회사」 라 합니다)의 승낙으로 보험계 약에 부가하여 이루어집니다. ② '
 '제1조(적용대상)의 보험계약이 해지 또는 기타 사유에 의하여 효력이 없게 된 경우에 는 이 특별약관은 더 이상 효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000849',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
