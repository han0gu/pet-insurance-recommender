from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.\n'
 '제4조 (특별약관의 무효)\n'
 '다음 중 한 가지에 해당되는 경우에는 이 특별약관은 무효로 하며 이미 납입한 이 특별약 관의 보험료를 돌려 드립니다. 다만, 회사의 고의 '
 '또는 과실로 특별약관이 무효로 된 경 우와 회사가 승낙 전에 무효임을 알았거나 알 수 있었음에도 불구하고 보험료를 반환하 지 않은 '
 '경우에는 보험료를 납입한 날의 다음날부터 반환일까지의 기간에 대하여 회사는 이 특별약관의 보험계약대출이율을 연단위 복리로 계산한 금액을 '
 '더하여 돌려 드립니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000692',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
