from langchain_core.documents import Document

chunk = Document(
    page_content=('- 산출방법서”에 따라 계산한 금액으로 해지율을 적용하지 않고 계산합니다.\n'
 '- • 회사는 계약을 체결할 때 표준형 상품의 보험료 및 해약환급금(환급률 포함) 수준을 비교∙안내해\n'
 '- 드립니다.\n'
 '- • 보험료 납입기간이란 계약을 체결할 때 보험료를 납입하기로 한 기간을 말합니다.\n'
 '4. 제1호, 제2호 및 제3호에도 불구하고 [갱신형] 특별약관 중 해약환급금 구분이 해'),
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
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
