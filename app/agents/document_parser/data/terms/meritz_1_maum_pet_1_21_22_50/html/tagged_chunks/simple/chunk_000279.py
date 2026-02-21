from langchain_core.documents import Document

chunk = Document(
    page_content=("위한 계약)</h1><br><p id='108' data-category='list' style='font-size:14px'>① "
 '계약자는 타인을 위한 계약을 체결하는 경우에 그 타인의 위임이 없는 때에는 반드시<br>이를 회사에 알려야 하며, 이를 알리지 않았을 '
 '때에는 그 타인은 이 계약이 체결된 사<br>실을 알지 못하였다는 사유로 회사에 이의를 제기할 수 없습니다.<br>② 타인을 위한 '
 '계약에서 보험사고가 발생한 경우에 계약자가 그 타인에게 보험사고의 발<br>생으로 생긴 손해를 배상한 때에는 계약자는 그 타인의 권리를'),
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
 'indexing': {'chunk_id': 'chunk_000279',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
