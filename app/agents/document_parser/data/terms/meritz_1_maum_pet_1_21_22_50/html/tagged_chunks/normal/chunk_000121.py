from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우<br>계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것으<br>로 봅니다.</p><br><p '
 "id='21' data-category='paragraph' style='font-size:14px'>【통신판매계약】</p><br><p "
 "id='22' data-category='paragraph' style='font-size:14px'>전화·우편·인터넷 등 통신수단을 "
 "이용하여 체결하는 계약을 말합니다.</p><br><p id='23' data-category='list'"),
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
 'indexing': {'chunk_id': 'chunk_000121',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
