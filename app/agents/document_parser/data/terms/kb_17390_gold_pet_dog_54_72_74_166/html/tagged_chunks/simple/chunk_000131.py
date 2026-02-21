from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한, 부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계<br>약을 최초계약으로 봅니다.</p><br><table '
 "id='155' style='font-size:16px'><thead></thead><tbody><tr><td "
 'colspan="2"></td></tr><tr><td colspan="2">유 의 사 항 보험계약을 청약하면서 보험설계사에게 질병이 '
 '있다고만 얘기하였을 뿐, 청약 공 서의 계약전 알릴 사항에 아무런 기재도 하지 않을 경우에는 보험설계사에게 통 병력을 얘기하였다고 '
 '하더라도 회사는 계약 전'),
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
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
