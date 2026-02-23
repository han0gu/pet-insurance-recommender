from langchain_core.documents import Document

chunk = Document(
    page_content=('- 산출방법서”에 따라 계산한 금액으로 해지율을 적용하지 않고 계산합니다.\n'
 '- • 회사는 계약을 체결할 때 표준형 상품의 보험료 및 해약환급금(환급률 포함) 수준을 비교∙안내해\n'
 '- 드립니다.\n'
 '- • 보험료 납입기간이란 계약을 체결할 때 보험료를 납입하기로 한 기간을 말합니다.\n'
 '4. 제1호, 제2호 및 제3호에도 불구하고 [갱신형] 특별약관 중 해약환급금 구분이 해\n'
 '약환급금 미지급형 및 해약환급금 미지급형Ⅱ을 제외한 경우에는 해당 특별약관의\n'
 '보험기간 중 계약이 해지될 경우 “보험료 및 해약환급금 산출방법서”에 따라 계산'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000273',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
