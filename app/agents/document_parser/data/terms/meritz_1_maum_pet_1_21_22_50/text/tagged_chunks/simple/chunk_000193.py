from langchain_core.documents import Document

chunk = Document(
    page_content=('가 보험금을 직접 청구할 수 없는 특별한 사정이 있음을 증명하는 서류를 제출하고 회\n'
 '사의 승낙을 얻어 제1조(적용대상)의 수익자의 대리인으로서 보험금(사망보험금 제외)\n'
 '을 청구하고 수령할 수 있습니다. 다만, 2인의 지정대리청구인이 지정된 경우에는 그\n'
 '중 대표대리인이 보험금을 청구하고 수령할 수 있으며, 대표대리인이 사망 등의 사유로\n'
 '보험금 청구가 불가능한 경우에는 대표가 아닌 지정대리청구인도 보험금을 청구하고 수\n'
 '령할 수 있습니다.\n'
 '② 회사가 보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더라도'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000193',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
