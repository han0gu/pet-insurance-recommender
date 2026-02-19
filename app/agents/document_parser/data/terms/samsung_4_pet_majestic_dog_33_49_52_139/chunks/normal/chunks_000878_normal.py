from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (지정대리청구인의 변경지정)\n'
 '① 계약자는 이 특별약관의 계약체결 이후 다음의 서류를 제출하고 지정대리청구인을 변 경 지정할 수 있습니다. 이 경우 회사는 변경 지정을 '
 '서면으로 알리거나 보험증권의 뒷면에 기재하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 139},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000878',
              'chunk_char_len': 127,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
