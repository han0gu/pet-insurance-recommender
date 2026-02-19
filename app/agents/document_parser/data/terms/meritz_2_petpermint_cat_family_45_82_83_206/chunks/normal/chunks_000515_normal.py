from langchain_core.documents import Document

chunk = Document(
    page_content=('【자기공명영상(MRI)】\n'
 '강한 자기장 내에서 반려동물에 고주파를 전사해서 반향 되는 전자기파를 측정하여 영상을 얻어 질병을 진단하는 검사\n'
 '【전산화단층영상(CT)】\n'
 'X선을 이용하여 반려동물의 횡단면상의 영상을 획득하여 진단에 이용하는 검사\n'
 '【내시경】\n'
 '내장장기 또는 체강(體腔) 내부를 직접 볼 수 있게 만든 의료기구\n'
 '제5조(특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 152},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000515',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
