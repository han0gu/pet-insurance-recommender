from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 보험금감액법\n'
 '계약일부터 회사가 정하는 삭감기간 내에 보험계약의 규정에 정하는 상해 이외의 원인으로 보험계약의 보험금 지급사유가 발생하였을 경우에는 '
 '보험계약의 규정에 도 불구하고 계약을 체결할 때 정한 삭감기간에 따라 다음과 같이 보험금을 지급 합니다.\n'
 '경과기간 | 기준 | 삭감기간별 보험금지급비율\n'
 '1년 | 2년 | 3년 | 4년 | 5년\n'
 '1년미만 | 보험계약에 정한 지급보험금 | 50% | 30% | 25% | 20% | 15%\n'
 '1년이상 2년미만 | 60% | 50% | 40% | 30%'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000675',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
