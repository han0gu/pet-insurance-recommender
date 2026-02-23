from langchain_core.documents import Document

chunk = Document(
    page_content=('등을 수신하였을 때에는 해당 문서를 드린 것으로 봅니다.- 1. 서면교부\n'
 '- 2. 우편 또는 전자우편\n'
 '- 3. 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시\n'
 '\uf000 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특약만 포함한\n'
 '약관을 드리며, 전화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의\n'
 '방법으로 약관의 중요한 내용을 설명할 수 있습니다.\n'
 '1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의중요한 내용 등 계약을 체결하는 데 필요한 사항을 질문 '
 '또는 설명하는 방법.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000112',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
