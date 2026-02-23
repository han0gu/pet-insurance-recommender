from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전화(음성녹\n'
 '음)로 안내하고자 할 때 다음 각 호의 요건을 모두 충족하100는 경우에「보험업감독규정」제4-36조 제3항에 따른 전자적\n'
 '상품설명장치를 활용할 수 있습니다.- ① 계약자에게 전자적 상품설명장치를 활용하여 제1항에\n'
 '- 따른 납입최고(독촉) 등을 한다는 사실을 미리 안내하\n'
 '- 고 동의를 받을 것\n'
 '- ② 전자적 상품설명장치를 활용하여 안내한 납입최고(독\n'
 '- 촉) 등을 계약자가 모두 수신하고 이해하였음을 확인\n'
 '- 할 것\n'
 '- ③ 계약자가 질의를 하거나 추가적인 설명을 요청하는 등'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
