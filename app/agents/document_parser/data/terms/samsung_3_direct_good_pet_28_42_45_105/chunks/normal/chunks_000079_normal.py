from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 우편 또는 전자우편 3. 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시\n'
 '② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특별약관만 포함한 약관을 드리며, 전화를 이용하여 체결하는 계약은 '
 '계약자의 동의를 얻어 다음의 방법 으로 약관의 중요한 내용을 설명할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000079',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
