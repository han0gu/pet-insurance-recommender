from langchain_core.documents import Document

chunk = Document(
    page_content=("체결하는 계약은 계약자의 동의를 얻어 다음의 방법으<br>로 약관의 중요한 내용을 설명할 수 있습니다.</p><br><p id='88' "
 "data-category='paragraph' style='font-size:20px'>① 전화를 이용하여 청약내용, 보험료납입, "
 "보험기간, 계</p><footer id='89' style='font-size:14px'>65</footer><h1 id='90' "
 "style='font-size:16px'>약 전 알릴 의무, 약관의 중요한 내용 등 계약을 체결<br>하는 데 필요한 사항을 질문 또는"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
