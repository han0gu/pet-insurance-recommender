from langchain_core.documents import Document

chunk = Document(
    page_content=('금 지급을 위해 필요하다고 인정하는 경우 관련 서류를 요\n'
 '청할 수 있습니다.88【동물병원 보험금 자동청구】지정된 동물병원에서 펫퍼민트 ID카드를 제시하고 진료\n'
 '를 받은 경우, 반려동물 치료비 결제 시에 보험금이 당\n'
 '사로 자동 청구되는 절차를 말합니다.# 제5조(보험금의 지급절차)\uf000 회사는 제4조(보험금의 청구)에서 정한 서류를 접수한\n'
 '때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우\n'
 '편 등으로도 송부하며, 그 서류를 접수한 날부터 3영업일\n'
 '이내에 보험금을 지급합니다.\n'
 '\uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
