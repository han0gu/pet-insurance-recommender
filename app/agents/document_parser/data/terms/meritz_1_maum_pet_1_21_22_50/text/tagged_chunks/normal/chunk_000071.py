from langchain_core.documents import Document

chunk = Document(
    page_content=('한 내용 등 계약을 체결하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우\n'
 '계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것으\n'
 '로 봅니다.【통신판매계약】전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.③ 회사가 제1항에 따라 제공될 약관 및 '
 '계약자 보관용 청약서를 청약할 때 계약자에게\n'
 '전달하지 않거나 약관의 중요한 내용을 설명하지 않은 때 또는 계약을 체결할 때 계약\n'
 '자가 청약서에 자필서명(날인(도장을 찍음) 및 ⌜전자서명법⌟ 제2조 제2호에 따른 전'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
