from langchain_core.documents import Document

chunk = Document(
    page_content=('. 의료기관의 종별은 종합병원・병원・ 치과병원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산 원으로 '
 "나누어집니다.</td></tr></tbody></table><br><h1 id='20' "
 "style='font-size:14px'>제8조(특별약관의 소멸)</h1><br><h1 id='21' "
 "style='font-size:14px'>피보험자가 사망하였을</h1><br><p id='22' "
 "data-category='paragraph' style='font-size:14px'>경우에는 이 특별약관의 계약도 소멸되며 회사는"),
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
 'indexing': {'chunk_id': 'chunk_000553',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
