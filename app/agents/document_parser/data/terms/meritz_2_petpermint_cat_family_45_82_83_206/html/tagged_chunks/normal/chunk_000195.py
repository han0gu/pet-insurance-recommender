from langchain_core.documents import Document

chunk = Document(
    page_content=("회사가<br>계약자에게 납입을 재촉하는 것을 말합니다.</p><br><p id='61' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전자문서로<br>안내하고자 할 경우에는 "
 '계약자에게 서면 또는 전자서명법<br>제2조 제2호에 따른 전자서명으로 동의를 얻어 수신확인을<br>조건으로 전자문서를 송신하여야 하며, '
 '계약자가 전자문서<br>에 대하여 수신을 확인하기 전까지는 그 전자문서는 송신되<br>지 않은 것으로 봅니다'),
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
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
