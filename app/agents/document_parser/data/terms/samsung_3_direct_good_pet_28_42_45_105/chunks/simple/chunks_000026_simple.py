from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류(단, 단체취급 특별약 관을 부가하는 경우, 사망보험금을 지급할 때 피보험자의 '
 '법정상속인이 아닌 자가 청구하는 경우 법정상속인의 확인서 등)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 30},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 111,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
