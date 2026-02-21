from langchain_core.documents import Document

chunk = Document(
    page_content=('하는 데 필요한 사항을 질문 또는 설명하는 방법. 이\n'
 '경우 계약자의 답변과 확인내용을 음성 녹음함으로써\n'
 '약관의 중요한 내용을 설명한 것으로 봅니다.# 【통신판매계약】전화·우편·인터넷 등 통신수단을 이용하여 체결하는\n'
 '계약을 말합니다.\uf000 회사가 제1항에 따라 제공될 약관 및 계약자 보관용 청\n'
 '약서를 청약할 때 계약자에게 전달하지 않거나 약관의 중요\n'
 '한 내용을 설명하지 않은 때 또는 계약을 체결할 때 계약자\n'
 '가 청약서에 자필서명을 하지 않은 때에는 계약자는 계약이'),
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
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
