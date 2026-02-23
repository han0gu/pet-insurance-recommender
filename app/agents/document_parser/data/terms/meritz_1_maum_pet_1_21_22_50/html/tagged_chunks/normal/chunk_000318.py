from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제4조(보험의 목적의 증가 감소 또는 교체)</h1><br><p id='66' "
 "data-category='list' style='font-size:14px'>① 계약을 맺은 후 보험의 목적을 증가, 감소 또는 "
 '교체코자 하는 경우에는 계약자 또는 피<br>보험자는 지체없이 서면으로 그 사실을 회사에 알리고 회사의 승인을 받아야 합니다.<br>② '
 '이 계약기간 중 보험의 목적 감소의 경우는 당해 보험의 목적의 계약은 해지된 것으로<br>하며 새로이 증가 또는 교체되는 보험의 목적의 '
 '보험기간은 이'),
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
 'indexing': {'chunk_id': 'chunk_000318',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
