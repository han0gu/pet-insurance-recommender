from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 저축성 보험계약의 공시이율\n'
 '- - 유배당 보험계약의 경우 계약자 배당에 관한 사항\n'
 '- - 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '\uf000 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자\n'
 '가 가입한 특별약관만 포함한 약관을 드리며, 전화를 이용\n'
 '하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법으\n'
 '로 약관의 중요한 내용을 설명할 수 있습니다.① 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계69# 약 전 알릴 의무, 약관의 '
 '중요한 내용 등 계약을 체결\n'
 '하는 데 필요한 사항을 질문 또는 설명하는 방법. 이'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
