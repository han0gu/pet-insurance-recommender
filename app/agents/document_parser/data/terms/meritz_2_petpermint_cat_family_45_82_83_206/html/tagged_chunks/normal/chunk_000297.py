from langchain_core.documents import Document

chunk = Document(
    page_content=('동물병원에서 펫퍼민트 ID카드를 제시하고 진료<br>를 받은 경우, 반려동물 치료비 결제 시에 보험금이 당<br>사로 자동 청구되는 '
 "절차를 말합니다.</p><h1 id='25' style='font-size:20px'>제5조(보험금의 지급절차)</h1><br><p "
 "id='26' data-category='paragraph' style='font-size:16px'>\uf000 회사는 제4조(보험금의 "
 '청구)에서 정한 서류를 접수한<br>때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우<br>편 등으로도 송부하며, 그 서류를 접수한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000297',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
