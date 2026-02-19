from langchain_core.documents import Document

chunk = Document(
    page_content=('【상법 제731조(타인의 생명의 보험)】\n'
 '① 타인의 사망을 보험사고로 하는 보험계약에는 보험계 약 체결시에 그 타인의 서면(｢전자서명법｣제2조제2호에 따른 전자서명이 있는 '
 '경우로서 대통령령으로 정하는 바 에 따라 본인확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함한다)에 의한 동의를 얻어야 '
 '한다.\n'
 '【상법 시행령 제44조의2(타인의 생명보험)】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 67},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 200,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
