from langchain_core.documents import Document

chunk = Document(
    page_content=('57 / 181\n'
 '5. 제1호, 제2호 및 제3호에서 표준형 상품이란 보험료 산출시 적용한 모든 기초율(다 만, 해지율은 적용하지 않습니다)이 동일한 '
 '상품을 말하며, 해약환급금을 계산할 때 기준이 되거나 비교∙안내를 위한 상품으로서 판매는 하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000292',
              'chunk_char_len': 141,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
