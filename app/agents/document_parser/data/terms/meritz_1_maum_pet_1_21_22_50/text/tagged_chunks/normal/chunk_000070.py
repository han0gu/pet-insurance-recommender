from langchain_core.documents import Document

chunk = Document(
    page_content=('때에는 해당 문서를 드린 것으로 봅니다.1. 서면교부\n'
 '2. 우편 또는 전자우편\n'
 '3. 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특약만 포함한 '
 '약관을\n'
 '드리며, 전화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법으로 약관\n'
 '의 중요한 내용을 설명할 수 있습니다.1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요\n'
 '한 내용 등 계약을 체결하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
