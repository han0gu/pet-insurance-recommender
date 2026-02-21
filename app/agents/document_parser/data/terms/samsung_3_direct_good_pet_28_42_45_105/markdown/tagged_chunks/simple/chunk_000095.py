from langchain_core.documents import Document

chunk = Document(
    page_content=('- 녹음)로 다시 알려 드립니다.\n'
 '- ⑥ 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹음)로 안내하고자 할 때 다음\n'
 '- 간 ㅎ이 요거음 모두 충족하는 경우에 「보험언감독규정 제4-36조 제3항에 따른\n'
 '- 38 -# 다는 사실을 미리 안내하고 동의를 받을 것- 2. 전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두 '
 '수신\n'
 '- 하고 이해하였음을 확인할 것\n'
 '- 3. 계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의 활'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
