from langchain_core.documents import Document

chunk = Document(
    page_content=('제11조(손해방지의무)\n'
 '보험사고가 생긴 때에는 계약자 또는 피보험자는 손해의 방지와 경감에 힘써야 합니다. 만약, 계약자 또는 피보험자가 고의 또는 중대한 '
 '과실로 이를 게을리한 때에는 방지 또는 경감할 수 있었을 것으로 밝혀진 값을 손해액에서 뺍니다.\n'
 '제3관 계약자의 계약 전 알릴 의무 등\n'
 '제12조(계약 전 알릴 의무)\n'
 '계약자, 피보험자 또는 이들의 대리인은 청약할 때 청약서(질문서를 포함합니다)에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 '
 '알려야 합니다.\n'
 '【관련법규】'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000038',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
