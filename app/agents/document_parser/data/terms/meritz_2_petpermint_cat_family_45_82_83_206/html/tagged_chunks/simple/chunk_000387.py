from langchain_core.documents import Document

chunk = Document(
    page_content=('등을 계약자가 모두 수신하고 이해하였음을 확인<br>할 것<br>③ 계약자가 질의를 하거나 추가적인 설명을 요청하는 등<br>전자적 '
 '상품설명장치의 활용을 중단할 것을 요구하는<br>경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1<br>항에 따른 납입최고(독촉) '
 '등을 실시할 것<br>④ 전자적 상품설명장치에 안내의 속도와 음량을 조절할<br>수 있는 기능을 갖출 것<br>⑤ 제3호 및 제4호의 '
 "내용에 관한 사항을 계약자에게 안<br>내할 것</p><br><p id='38' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000387',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
