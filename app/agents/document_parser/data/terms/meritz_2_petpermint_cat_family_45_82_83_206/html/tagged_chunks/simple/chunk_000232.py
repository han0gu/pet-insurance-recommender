from langchain_core.documents import Document

chunk = Document(
    page_content=('. 중도인<br>출금의 총 누적액(중도인출 원금과 이자의 합계액을 말합니<br>다)은 중도인출금을 한번도 지급하지 않았을 경우의 '
 '기본계<br>약 해약환급금과 기본계약 적립부분 해약환급금 중 적은 금<br>액의 80%를 한도로 합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000232',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
