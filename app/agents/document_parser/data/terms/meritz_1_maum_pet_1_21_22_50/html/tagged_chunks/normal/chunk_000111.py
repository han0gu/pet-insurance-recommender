from langchain_core.documents import Document

chunk = Document(
    page_content=('자로서, 국가, 지방자치단체, 한국은행, 금융회사, 주권상장법<br>인 등을 포함하며 「금융소비자보호에 관한 법률」제2조(정의) '
 "제9호에서 정하는 전<br>문금융소비자를 말합니다.</p><br><p id='12' data-category='list' "
 "style='font-size:14px'>② 제1항에도 불구하고 청약한 날부터 30일이 초과된 계약은 청약을 철회할 수 "
 '없습니다.<br>③ 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우편, 휴<br>대전화 문자메시지 또는 '
 '이에 준하는 전자적'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000111',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
