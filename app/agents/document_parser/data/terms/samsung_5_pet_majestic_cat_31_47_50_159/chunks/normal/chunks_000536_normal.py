from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '「반려묘 의료비(치과및 구강질환포함)(재가입형)」 에 대한 보장개시일(책임개시일) 계산]\n'
 '보험계약일\n'
 '보장개시일(책임개시일)\n'
 '30일\n'
 '2022년 8월 1일\n'
 '2022년 8월 31일\n'
 '주) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 합니 다.\n'
 '④ 회사가 지급할 제1항에서 정한 의료비보험금은 보험증권에 기재된 자기부담금을 차감 한 후 보상비율을 곱한 금액이며 보험증권에 기재된 '
 '1일당 보상한도액을 한도로 합니 다. (자기부담금은 1일당 의료비에서 차감합니다)\n'
 '<지급보험금의 계산>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 97},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000536',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
