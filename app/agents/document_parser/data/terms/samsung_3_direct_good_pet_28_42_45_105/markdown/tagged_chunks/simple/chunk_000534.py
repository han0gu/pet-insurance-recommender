from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우는 제외합니다.)\n'
 '# 제3조 (갱신계약의 보험계약 적용 특칙)제2조에 따라 갱신된 갱신계약의 경우 아래에 정한 사항을 따릅니다.# 제도 또는 보험료(이하 '
 '「보험요율 제도 또는 보험료」 라 합니다)는 갱신일 현재의\n'
 '보험요율 제도 또는 보험료를 적용합니다. 단, 법령 및 표준약관의 제·개정 또는\n'
 '금융위원회의 명령에 따라 약관이 개정된 경우에는 갱신일 현재의 약관을 적용합\n'
 '니다.# 2. 갱신시 보험기간의 운영- 가. 갱신계약의 보험기간은 갱신전 계약의 보험기간과 동일하게 적용하며, 갱신계'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000534',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
