from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 제3항에도 불구하고 전화를 이용하여 계약을 체결하는 경우 다음의 각 호의 어느 하 나를 충족하는 때에는 자필서명을 생략할 수 있으며, '
 '제2항의 규정에 따른 음성녹음 내용을 문서화한 확인서를 계약자에게 드림으로써 계약자 보관용 청약서를 전달한 것 으로 봅니다.\n'
 '1. 계약자, 피보험자 및 보험수익자가 동일한 계약의 경우 2. 계약자, 피보험자가 동일하고 보험수익자가 계약자의 법정상속인인 계약일 '
 '경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
