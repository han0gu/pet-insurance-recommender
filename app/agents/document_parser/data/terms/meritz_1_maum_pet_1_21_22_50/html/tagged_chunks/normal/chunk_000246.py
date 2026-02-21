from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호사비용<br>과 회사의 동의를 받지 않은 행위로 증가된 '
 "손해</p><h1 id='74' style='font-size:14px'>제12조(손해배상청구에 대한 회사의 해결)</h1><br><p "
 "id='75' data-category='list' style='font-size:14px'>① 피보험자가 피해자에게 손해배상책임을 지는 "
 '사고가 생긴 때에는 피해자는 이 특별약관<br>에 따라 회사가 피보험자에게 지급책임을 지는 금액 한도내에서 회사에 대하여'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000246',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
