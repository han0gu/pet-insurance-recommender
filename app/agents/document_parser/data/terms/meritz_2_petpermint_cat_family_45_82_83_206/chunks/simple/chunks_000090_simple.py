from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자 가 가입한 특별약관만 포함한 약관을 드리며, 전화를 이용 하여 체결하는 '
 '계약은 계약자의 동의를 얻어 다음의 방법으 로 약관의 중요한 내용을 설명할 수 있습니다.\n'
 '① 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 65},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
