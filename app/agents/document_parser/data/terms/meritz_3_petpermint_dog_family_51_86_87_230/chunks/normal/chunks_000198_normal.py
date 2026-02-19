from langchain_core.documents import Document

chunk = Document(
    page_content=('【동물병원 보험금 자동청구】\n'
 '지정된 동물병원에서 펫퍼민트 ID카드를 제시하고 진료 를 받은 경우, 반려동물 치료비 결제 시에 보험금이 당 사로 자동 청구되는 절차를 '
 '말합니다.\n'
 '제5조(보험금의 지급절차)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000198',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
