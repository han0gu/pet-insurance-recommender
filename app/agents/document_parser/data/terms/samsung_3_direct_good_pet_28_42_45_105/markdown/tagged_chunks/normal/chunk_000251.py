from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 제1호, 제2호 및 제3호에도 불구하고 [갱신형] 특별약관 중 해약환급금 구분이 해\n'
 '약환급금 미지급형 및 해약환급금 미지급형Ⅱ을 제외한 경우에는 해당 특별약관의보험기간 중 계약이 해지될 경우 “보험료 및 해약환급금 '
 '산출방법서”에 따라 계산한 금액을 해약환급금으로 지급합니다.- 57 -57 / 1815. 제1호, 제2호 및 제3호에서 표준형 상품이란 '
 '보험료 산출시 적용한 모든 기초율(다\n'
 '만, 해지율은 적용하지 않습니다)이 동일한 상품을 말하며, 해약환급금을 계산할'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
