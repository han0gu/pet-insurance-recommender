from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 제1호, 제2호 및 제3호에도 불구하고 [갱신형] 특별약관 중 해약환급금 구분이 해 약환급금 미지급형 및 해약환급금 미지급형Ⅱ을 '
 '제외한 경우에는 해당 특별약관의\n'
 '보험기간 중 계약이 해지될 경우 “보험료 및 해약환급금 산출방법서”에 따라 계산\n'
 '한 금액을 해약환급금으로 지급합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 57},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000291',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
