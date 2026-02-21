from langchain_core.documents import Document

chunk = Document(
    page_content=('를 연대로 합니다.<용어풀이># [연대]2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되(지분만큼 분할하여\n'
 '책임을 지는 것과 다름), 어느 1인의 이행으로 나머지 사람들도 책임을 면하게 되는 것을 말합니\n'
 '다.제3관 계약자의 계약 전 알릴 의무 등# 제 15조 (계약 전 알릴 의무)계약자 또는 피보험자는 청약할 때(진단계약의 경우에는 '
 '건강진단할 때를 말합니다) 청약\n'
 '서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하「계약 전'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
